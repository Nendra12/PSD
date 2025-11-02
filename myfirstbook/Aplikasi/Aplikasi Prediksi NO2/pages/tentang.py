import streamlit as st

st.set_page_config(page_title="Tentang Aplikasi", page_icon="../icon/icon.png", layout="centered")

st.title("Tentang Aplikasi NO₂ Quality Detector")
if st.button("Kembali ke Beranda"):
    st.switch_page("app_main.py")

st.markdown("""
### Tentang Aplikasi
Aplikasi ini bernama **NO₂ Quality Detector (KNN)** — sebuah sistem **prediksi kualitas udara berbasis Machine Learning (K-Nearest Neighbors)**.  
Model ini memprediksi **konsentrasi gas Nitrogen Dioksida (NO₂)** untuk **hari esok (t+1)** dan **dua hari mendatang (t+2)** berdasarkan tren 5 hari terakhir.  

Data diambil dari **Copernicus Open Data (Sentinel-5P L2 Collection)** Data Sentinel-5P merekam kandungan **NO₂ troposferik** di atmosfer dengan resolusi tinggi, cocok untuk pemantauan udara di **Kota Semarang**.

---

### Tentang Gas NO₂
**Nitrogen Dioksida (NO₂)** adalah polutan utama dari:
- Emisi kendaraan bermotor  
- Industri dan pembangkit listrik  
- Aktivitas rumah tangga seperti pembakaran sampah  

Paparan tinggi menyebabkan **iritasi saluran pernapasan**, **penurunan fungsi paru**, dan memperburuk **asma**.

---

### Format dan Satuan Data
Data Sentinel-5P disajikan dalam satuan:
> **mol/m² (molekul NO₂ per meter persegi)**

Model ini menggunakan **data yang telah dinormalisasi menggunakan min-max scaler** agar konsisten untuk prediksi dan perbandingan antar waktu.

---

### Cara Kerja Aplikasi
1. Masukkan nilai NO₂ 5 hari terakhir  
2. Sistem memprediksi kadar NO₂ untuk 1 atau 2 hari ke depan menggunakan model **KNN Regression**  
3. Hasil dibandingkan dengan **threshold WHO**  
   - ≤ threshold → **Baik (🟢)**  
   - > threshold → **Buruk (🔴)**

---

### Standar WHO untuk NO₂
Menurut pedoman **WHO (2021)**:
- Maksimum tahunan: **10 µg/m³**
- Maksimum per jam: **25 µg/m³**

Perkiraan ekuivalen dalam data Sentinel-5P:
> **2.5×10⁻⁵ – 5.0×10⁻⁵ mol/m²**

Threshold default:
> **0.000050 (5.0×10⁻⁵ mol/m²)**

---

### 🧩 Interpretasi Hasil
| Kategori | Arti | Rekomendasi |
|-----------|------|-------------|
| 🟢 **Baik** | Kadar NO₂ rendah | Aman untuk aktivitas luar ruangan |
| 🔴 **Buruk** | Kadar NO₂ tinggi | Kurangi aktivitas luar, terutama bagi anak dan lansia |

---

### 💬 Catatan
Model ini menggunakan **multi-output regression**, sehingga mampu memprediksi beberapa hari ke depan (multi-day forecast).  
Hasil bersifat **prediktif**, bukan pengukuran langsung, dan disarankan untuk dikombinasikan dengan **data sensor lokal** untuk akurasi lebih tinggi.
""")

if st.button("Prediksi Sekarang!"):
    st.switch_page("app_main.py")