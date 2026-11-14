import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils.prosesing import SKY_CMAP, TARGET, iqr_bounds, outlier_summary, require_dataset, setup_page, split_columns

setup_page("Exploratory Data Analysis", "🔍", subtitle="Kategorikal, numerik, outlier, serta hubungan antar atribut")

df = require_dataset()
num_cols, cat_cols = split_columns(df)


def index_of(options, name, default=0):
    return options.index(name) if name in options else default


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


tab1, tab2, tab3, tab4 = st.tabs(
    ["Kategorikal", "Numerik", "Outlier (IQR)", "Univariate & Bivariate"]
)

# ---------------------------------------------------------------- 5.1 Kategorikal
with tab1:
    st.subheader("Analisis Atribut Kategorikal")
    if not cat_cols:
        st.info("Dataset tidak memiliki atribut kategorikal.")
    else:
        col = st.selectbox("Pilih atribut kategorikal", cat_cols, key="cat_col")
        counts = df[col].value_counts(dropna=False)
        left, right = st.columns([2, 1])
        with left:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(counts.index.astype(str), counts.values, color="#7FA8D6")
            ax.bar_label(bars)
            ax.set_xlabel(col)
            ax.set_ylabel("Jumlah siswa")
            ax.set_title(f"Distribusi {col}")
            show(fig)
        with right:
            table = pd.DataFrame(
                {
                    "Kategori": counts.index.astype(str),
                    "Jumlah": counts.values,
                    "Persentase (%)": (counts.values / len(df) * 100).round(2),
                }
            )
            st.dataframe(table, hide_index=True)

# ---------------------------------------------------------------- 5.2 Numerik
with tab2:
    st.subheader("Analisis Atribut Numerik")
    col = st.selectbox("Pilih atribut numerik", num_cols, index=index_of(num_cols, TARGET), key="num_col")
    s = df[col].dropna()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean", f"{s.mean():.2f}")
    m2.metric("Median", f"{s.median():.2f}")
    m3.metric("Minimum", f"{s.min():.2f}")
    m4.metric("Maximum", f"{s.max():.2f}")
    m5.metric("Std. Deviasi", f"{s.std():.2f}")

    bins = st.slider("Jumlah bin histogram", 5, 50, 20, key="num_bins")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=bins, color="#7FA8D6", edgecolor="#FFFDF2")
    ax.axvline(s.mean(), color="#2B4162", linestyle="--", label=f"Mean = {s.mean():.2f}")
    ax.axvline(s.median(), color="#E9B949", linestyle="-.", label=f"Median = {s.median():.2f}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frekuensi")
    ax.set_title(f"Histogram {col}")
    ax.legend()
    show(fig)

# ---------------------------------------------------------------- 5.3 Outlier
with tab3:
    st.subheader("Deteksi Outlier dengan IQR")
    st.latex(r"IQR = Q_3 - Q_1")
    st.latex(r"Lower = Q_1 - 1.5 \times IQR \qquad Upper = Q_3 + 1.5 \times IQR")

    default = [c for c in ["absences", "age", "G1", "G2", "G3"] if c in num_cols]
    selected = st.multiselect("Atribut yang dianalisis", num_cols, default=default, key="out_cols")

    if not selected:
        st.info("Pilih minimal satu atribut numerik.")
    else:
        st.dataframe(outlier_summary(df, selected).round(2), hide_index=True)

        col = st.selectbox("Tampilkan boxplot untuk", selected, key="out_box")
        q1, q3, iqr, lower, upper = iqr_bounds(df[col])
        s = df[col].dropna()
        mask = (s < lower) | (s > upper)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot(s)
        ax.set_xticklabels([col])
        ax.set_title(f"Boxplot {col} — {int(mask.sum())} outlier")
        show(fig)

        with st.expander(f"Lihat baris outlier pada {col}"):
            if mask.sum() == 0:
                st.write("Tidak ada outlier.")
            else:
                st.dataframe(df.loc[s[mask].index])

        st.info(
            "ℹ️ Outlier **tidak dihapus** — hanya dideteksi. Nilai absensi yang tinggi, "
            "misalnya, bisa merupakan kondisi siswa yang benar-benar terjadi."
        )

# ---------------------------------------------------------------- 5.4 Uni / Bivariate
with tab4:
    mode = st.radio(
        "Jenis analisis",
        [
            "Univariate",
            "Bivariate: numerik vs numerik",
            "Bivariate: kategorikal vs numerik",
            "Correlation matrix",
        ],
        horizontal=True,
    )

    if mode == "Univariate":
        all_cols = list(df.columns)
        col = st.selectbox("Pilih atribut", all_cols, index=index_of(all_cols, "age"), key="uni_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        if col in num_cols:
            s = df[col].dropna()
            sns.histplot(s, bins=20, kde=s.nunique() > 1, ax=ax, color="#7FA8D6")
            ax.set_title(f"Distribusi {col}")
        else:
            counts = df[col].value_counts()
            bars = ax.bar(counts.index.astype(str), counts.values, color="#7FA8D6")
            ax.bar_label(bars)
            ax.set_title(f"Distribusi {col}")
        show(fig)

    elif mode == "Bivariate: numerik vs numerik":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Sumbu X", num_cols, index=index_of(num_cols, "absences"), key="bi_x")
        y = c2.selectbox("Sumbu Y", num_cols, index=index_of(num_cols, TARGET), key="bi_y")
        data = df[[x, y]].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data[x], data[y], alpha=0.5, color="#7FA8D6")
        if data[x].nunique() > 1:
            slope, intercept = np.polyfit(data[x], data[y], 1)
            xs = np.linspace(data[x].min(), data[x].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="#E9B949", linewidth=2.5, label="Garis regresi")
            ax.legend()
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        show(fig)
        if x != y and data[x].nunique() > 1 and data[y].nunique() > 1:
            r = data[x].corr(data[y])
            st.metric("Korelasi Pearson (r)", f"{r:.3f}")

    elif mode == "Bivariate: kategorikal vs numerik":
        if not cat_cols:
            st.info("Dataset tidak memiliki atribut kategorikal.")
        else:
            c1, c2 = st.columns(2)
            cat = c1.selectbox("Atribut kategorikal", cat_cols, key="bc_cat")
            num = c2.selectbox("Atribut numerik", num_cols, index=index_of(num_cols, TARGET), key="bc_num")
            groups = sorted(df[cat].dropna().unique(), key=str)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot([df.loc[df[cat] == g, num].dropna() for g in groups])
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([str(g) for g in groups])
            ax.set_xlabel(cat)
            ax.set_ylabel(num)
            ax.set_title(f"{num} berdasarkan {cat}")
            show(fig)
            st.dataframe(df.groupby(cat)[num].agg(["count", "mean", "median", "std"]).round(2))

    else:  # correlation matrix
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=SKY_CMAP, center=0, ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation Matrix (atribut numerik)")
        show(fig)
        if TARGET in corr.columns:
            st.markdown(f"**Korelasi dengan {TARGET}**")
            st.dataframe(
                corr[TARGET].drop(TARGET).sort_values(key=lambda v: v.abs(), ascending=False).round(3).to_frame("Korelasi")
            )
