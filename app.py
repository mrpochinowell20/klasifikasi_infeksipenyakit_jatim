
import streamlit as st
import pandas as pd
import numpy as np
import jenkspy
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Dashboard Klasifikasi DBD Berdasarkan Jumlah Kasus")

# Upload data CSV
uploaded_file = st.file_uploader("Unggah file CSV hasil bersih:", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    data = df["jumlah_kasus"].values

    # Jenks klasifikasi
    breaks = jenkspy.jenks_breaks(data, nb_class=3)

    def klasifikasi(val):
        if val <= breaks[1]:
            return "rendah"
        elif val <= breaks[2]:
            return "sedang"
        else:
            return "tinggi"

    df["klaster"] = df["jumlah_kasus"].apply(klasifikasi)

    # SDCM & SDAM
    sdcm_dict = {}
    sdcm_total = 0
    for k in df["klaster"].unique():
        grp = df[df["klaster"] == k]["jumlah_kasus"]
        mean = grp.mean()
        s = np.sum((grp - mean) ** 2)
        sdcm_dict[k] = s
        sdcm_total += s

    sdam = np.sum((data - np.mean(data)) ** 2)
    sc = sdcm_total / sdam if sdam != 0 else 0

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Data Klaster")
        st.dataframe(df)

    with col2:
        st.subheader("📈 Statistik Klasifikasi")
        st.metric("Total SDCM", f"{sdcm_total:.2f}")
        st.metric("Total SDAM", f"{sdam:.2f}")
        st.metric("Koef. Efisiensi (SC)", f"{sc:.4f}")

    # Grafik pie
    st.subheader("🧩 Distribusi Klaster")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Distribusi Wilayah per Klaster**")
        fig1, ax1 = plt.subplots()
        ax1.pie(
            df["klaster"].value_counts(),
            labels=df["klaster"].value_counts().index,
            autopct='%1.1f%%',
            colors=["green", "orange", "red"]
        )
        st.pyplot(fig1)

    with col4:
        st.markdown("**Distribusi Jumlah Kasus per Klaster**")
        fig2, ax2 = plt.subplots()
        ax2.pie(
            df.groupby("klaster")["jumlah_kasus"].sum(),
            labels=df.groupby("klaster")["jumlah_kasus"].sum().index,
            autopct='%1.1f%%',
            colors=["green", "orange", "red"]
        )
        st.pyplot(fig2)

    # Grafik SDCM batang
    st.subheader("📊 Grafik SDCM per Klaster")
    fig3, ax3 = plt.subplots()
    ax3.bar(sdcm_dict.keys(), sdcm_dict.values(), color=["green", "orange", "red"])
    ax3.set_xlabel("Klaster")
    ax3.set_ylabel("Nilai SDCM")
    ax3.set_title("SDCM per Klaster")
    st.pyplot(fig3)
