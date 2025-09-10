# Data Understanding

> **Apa isi dataset ini?**  
> Dataset iris terdiri dari **150 baris data** dan **5 kolom utama**.  
> Empat kolom berisi pengukuran bunga, sedangkan satu kolom lagi berisi label spesies.  

---

## 📂 Struktur Data
- **Sepal Length** 🌿 → Panjang kelopak bunga (dalam cm)  
- **Sepal Width** 🌿 → Lebar kelopak bunga (dalam cm)  
- **Petal Length** 🌸 → Panjang mahkota bunga (dalam cm)  
- **Petal Width** 🌸 → Lebar mahkota bunga (dalam cm)  
- **Species** 🏷️ → Jenis bunga (*Setosa*, *Versicolor*, *Virginica*)  

---

## 🔎 Observasi Awal
- Jumlah data **150 baris**, terbagi rata:  
  - 50 data *Setosa*  
  - 50 data *Versicolor*  
  - 50 data *Virginica*  
- Data **sudah bersih** → tidak ada nilai kosong atau duplikat.  
- Ada perbedaan pola ukuran petal & sepal antar spesies,  
  misalnya *Setosa* cenderung punya petal yang paling kecil.  

---

## 📊 Analisis Tipe Data
| Kolom         | Tipe Data | Keterangan |
|---------------|-----------|------------|
| Sepal Length  | Float     | Data numerik kontinu (cm) |
| Sepal Width   | Float     | Data numerik kontinu (cm) |
| Petal Length  | Float     | Data numerik kontinu (cm) |
| Petal Width   | Float     | Data numerik kontinu (cm) |
| Species       | String    | Kategori (3 kelas: Setosa, Versicolor, Virginica) |

---

## 📈 Distribusi Data
| Spesies       | Jumlah Data |
|---------------|-------------|
| Setosa        | 50 |
| Versicolor    | 50 |
| Virginica     | 50 |

➡️ Distribusi sangat seimbang, sehingga dataset ini cocok untuk tugas klasifikasi karena tidak ada masalah *imbalanced data*.  

---

## 🧹 Kualitas Data
- **Konsistensi Data** ✅  
  Semua kolom sudah memiliki format konsisten: numerik untuk ukuran bunga dan string untuk label spesies.  

- **Missing Value** 🚫  
  Tidak ditemukan nilai kosong (*null*) pada dataset.  

- **Duplikasi Data** 🚫  
  Setiap baris merepresentasikan sampel bunga unik, tidak ada duplikat.  

- **Outlier** ⚠️  
  Secara umum, rentang nilai sepal & petal sesuai dengan literatur dataset Iris.  
  Namun, terdapat beberapa nilai ekstrem (contoh: sepal width = 2.0 atau 4.4) yang bisa dianggap *outlier* tapi masih dalam batas wajar untuk data biologis.  

- **Balance Class** ✅  
  Tiap kelas spesies memiliki jumlah data yang sama (50), sehingga tidak terjadi *class imbalance*.  

---

## 📑 Contoh 10 Data Awal
| Species      | Sepal Length | Sepal Width | Petal Length | Petal Width |
|--------------|--------------|-------------|--------------|-------------|
| Iris-setosa  | 5.1          | 3.5         | 1.4          | 0.2 |
| Iris-setosa  | 4.9          | 3.0         | 1.4          | 0.2 |
| Iris-setosa  | 4.7          | 3.2         | 1.3          | 0.2 |
| Iris-setosa  | 4.6          | 3.1         | 1.5          | 0.2 |
| Iris-setosa  | 5.0          | 3.6         | 1.4          | 0.2 |
| Iris-setosa  | 5.4          | 3.9         | 1.7          | 0.4 |
| Iris-setosa  | 4.6          | 3.4         | 1.4          | 0.3 |
| Iris-setosa  | 5.0          | 3.4         | 1.5          | 0.2 |
| Iris-setosa  | 4.4          | 2.9         | 1.4          | 0.2 |
| Iris-setosa  | 4.9          | 3.1         | 1.5          | 0.1 |

---

## 📌 Insight
Dataset ini cukup sederhana tapi informatif, sehingga cocok dipakai untuk:  
- Melatih model klasifikasi dasar.  
- Membandingkan performa berbagai algoritma machine learning.  
- Menjadi dataset latihan untuk eksplorasi visualisasi data.  
