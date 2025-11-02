import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Semarang NO₂ Prediction", page_icon="icon/icon.png", layout="centered")
st.title("Prediksi Kadar Nitrogen Dioksida (NO₂) di Kota Semarang Berbasis K-Nearest Neighbors")

if st.button("Baca Penjelasan Aplikasi Disini"):
    st.switch_page("pages/tentang.py")

st.caption("Masukkan NO₂ 5 hari terakhir → prediksi NO₂ hari esok, 2 hari lagi, atau keduanya.")

try:
    with open("model/model_knn_day5.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("model/scaler_day5.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Gagal memuat model/scaler: {e}")
    st.stop()

st.markdown("### 1) Input NO₂ 3 Hari Terakhir")
t1 = st.number_input("NO₂ (kemarin)", value=0.0, format="%.6f")
t2 = st.number_input("NO₂ (-2 hari)", value=0.0, format="%.6f")
t3 = st.number_input("NO₂ (-3 hari)", value=0.0, format="%.6f")
t4 = st.number_input("NO₂ (-4 hari)", value=0.0, format="%.6f")
t5 = st.number_input("NO₂ (-5 hari)", value=0.0, format="%.6f")


st.markdown("### 2) Pilih Mode Prediksi")
mode = st.radio(
    "Pilih jenis prediksi yang diinginkan:",
    ("Besok (t+1)", "2 Hari Lagi (t+2)", "Keduanya")
)

st.markdown("### 3) Threshold Kategori")
st.info("Jika Prediksi ≤ Threshold → **Baik**, jika > Threshold → **Buruk**.")
threshold = st.number_input("Threshold NO₂ (0.000217 mol/m²)", value=0.000217, format="%.6f")

if st.button("Prediksi", use_container_width=True):
    try:
        X_new = np.array([[t1, t2, t3, t4, t5]])
        X_new_scaled = scaler.transform(X_new)
        y_pred = knn.predict(X_new_scaled)[0]  
        pred_t1, pred_t2 = y_pred[0], y_pred[1]

        kategori_t1 = "Baik ✅" if pred_t1 <= threshold else "Buruk ⚠️"
        kategori_t2 = "Baik ✅" if pred_t2 <= threshold else "Buruk ⚠️"
        warna_t1 = "🟢" if kategori_t1.startswith("Baik") else "🔴"
        warna_t2 = "🟢" if kategori_t2.startswith("Baik") else "🔴"

        st.markdown("### Hasil Prediksi")

        if mode == "Besok (t+1)":
            st.metric("Prediksi NO₂ (Besok)", f"{pred_t1:.6f}")
            st.subheader(f"{warna_t1} Kategori: {kategori_t1}")
        
        elif mode == "2 Hari Lagi (t+2)":
            st.metric("Prediksi NO₂ (2 Hari Lagi)", f"{pred_t2:.6f}")
            st.subheader(f"{warna_t2} Kategori: {kategori_t2}")
        
        else: 
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediksi NO₂ (Besok / t+1)", f"{pred_t1:.6f}")
                st.subheader(f"{warna_t1} {kategori_t1}")
            with col2:
                st.metric("Prediksi NO₂ (2 Hari Lagi / t+2)", f"{pred_t2:.6f}")
                st.subheader(f"{warna_t2} {kategori_t2}")

        st.caption("Catatan: Nilai lebih tinggi dari threshold menunjukkan kualitas udara yang buruk.")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
