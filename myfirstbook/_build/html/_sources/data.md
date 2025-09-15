# Data Understanding

> **Apa isi dataset ini?**  
> Dataset iris terdiri dari **150 baris data** dan **5 kolom utama**.  
> Empat kolom berisi pengukuran bunga, sedangkan satu kolom lagi berisi label spesies.  

---

## Contoh 10 Data Iris Awal
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

## Struktur Data
- **Sepal Length** → Panjang kelopak bunga (dalam cm)  
- **Sepal Width** → Lebar kelopak bunga (dalam cm)  
- **Petal Length** → Panjang mahkota bunga (dalam cm)  
- **Petal Width** → Lebar mahkota bunga (dalam cm)  
- **Species** → Jenis bunga (*Setosa*, *Versicolor*, *Virginica*)  

---

## Observasi Awal
- Jumlah data **150 baris**, terbagi rata:  
  - 50 data *Setosa*  
  - 50 data *Versicolor*  
  - 50 data *Virginica*  
- Data **sudah bersih** → tidak ada nilai kosong atau duplikat.  
- Ada perbedaan pola ukuran petal & sepal antar spesies,  
  misalnya *Setosa* cenderung punya petal yang paling kecil.  

---

## Analisis Tipe Data
| Kolom         | Tipe Data | Keterangan |
|---------------|-----------|------------|
| Sepal Length  | Float     | Data numerik kontinu (cm) |
| Sepal Width   | Float     | Data numerik kontinu (cm) |
| Petal Length  | Float     | Data numerik kontinu (cm) |
| Petal Width   | Float     | Data numerik kontinu (cm) |
| Species       | String    | Kategori (3 kelas: Setosa, Versicolor, Virginica) |

---

## Distribusi Data
| Spesies       | Jumlah Data |
|---------------|-------------|
| Setosa        | 50 |
| Versicolor    | 50 |
| Virginica     | 50 |

Distribusi sudah seimbang, jadi dataset ini cocok untuk tugas klasifikasi karena tidak ada masalah *imbalanced data*.  

---

## Outlier Detection

Pada deteksi outlier saya menggunakan 3 metode antara lain ABOD, KNN, LOF :

### ABOD 

```python
import pandas as pd
from pycaret.anomaly import * 
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("iris.csv")
data = data.drop(columns=["iris"])

# Setup data PyCaret
s = setup(data)

# Buat model ABOD dengan fraction 5%
abod = create_model('abod', fraction=0.05) 
results = assign_model(abod)

# Urutkan data berdasarkan Anomaly 1 dan 0 Outlier
results_sorted = results.sort_values(by="Anomaly", ascending=False)

# Ringkasan hasil outlier detection (tampilkan 10 baris pertama)
print("Ringkasan Hasil Outlier Detection (10 baris pertama):")
print(results_sorted[['sepal length','sepal width','petal length','petal width','Anomaly', 'Anomaly_Score']].head(10))

# Hitung jumlah outlier dan normal
outlier_count = results['Anomaly'].sum()
normal_count = len(results) - outlier_count
total_count = len(results)
outlier_percent = (outlier_count / total_count) * 100

print("\n Statistik Outlier:")
print(f"Total Data     : {total_count}")
print(f"Normal Data    : {normal_count}")
print(f"Outlier Data   : {outlier_count}")
print(f"Persentase Outlier : {outlier_percent:.2f}%")

# Ringkasan dalam bentuk DataFrame
summary_df = pd.DataFrame({
    'Kategori': ['Normal', 'Outlier'],
    'Jumlah': [normal_count, outlier_count],
    'Persentase': [100 - outlier_percent, outlier_percent]
})
print("\n Ringkasan Jumlah Data:")
print(summary_df)

# Visualisasi scatter plot
plt.figure(figsize=(8,6))
plt.scatter(
    results['sepal length'], results['sepal width'],
    c=results['Anomaly'], cmap='coolwarm', edgecolor='k'
)

# Tambahkan judul dan label
plt.title("Visualisasi Deteksi Outlier dengan ABOD")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# Tambahkan legenda
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Normal',
               markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Outlier',
               markerfacecolor='red', markersize=8)
]
plt.legend(handles=handles)

plt.show()
```

#### Ringkasan Hasil Outlier Detection (10 baris pertama) ABOD

| Index | Sepal Length | Sepal Width | Petal Length | Petal Width | Anomaly | Anomaly Score |
|-------|--------------|-------------|--------------|-------------|---------|---------------|
| 131   | 7.9          | 3.8         | 6.4          | 2.0         | 1       | -0.137417     |
| 62    | 6.0          | 2.2         | 4.0          | 1.0         | 1       | -0.289391     |
| 108   | 6.7          | 2.5         | 5.8          | 1.8         | 1       | -0.084224     |
| 106   | 4.9          | 2.5         | 4.5          | 1.7         | 1       | -0.050388     |
| 117   | 7.7          | 3.8         | 6.7          | 2.2         | 1       | -0.129286     |
| 100   | 6.3          | 3.3         | 6.0          | 2.5         | 1       | -0.364475     |
| 134   | 6.1          | 2.6         | 5.6          | 1.4         | 1       | -0.309969     |
| 41    | 4.5          | 2.3         | 1.3          | 0.3         | 1       | -0.088998     |
| 84    | 5.4          | 3.0         | 4.5          | 1.5         | 0       | -3.127326     |
| 97    | 6.2          | 2.9         | 4.3          | 1.3         | 0       | -29.977802    |

---

#### Statistik Outlier ABOD

- **Total Data**     : 150  
- **Normal Data**    : 142  
- **Outlier Data**   : 8  
- **Persentase Outlier** : 5.33%  

---

#### Ringkasan Jumlah Data ABOD

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Normal   | 142    | 94.67%     |
| Outlier  | 8      | 5.33%      |

---

#### Visualisasi Outlier ABOD


![Petal Length](abod_2.png)

### KNN

```python
import pandas as pd
from pycaret.anomaly import * 
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")
data = data.drop(columns=["iris"])

s = setup(data)

knn = create_model('knn') 
results = assign_model(knn)

results_sorted = results.sort_values(by="Anomaly", ascending=False)

# Ringkasan hasil outlier detection (tampilkan 10 baris pertama)
print("Ringkasan Hasil Outlier Detection (10 baris pertama):")
print(results_sorted[['sepal length','sepal width','petal length','petal width','Anomaly','Anomaly_Score']].head(10))

# Hitung jumlah outlier dan normal
outlier_count = results['Anomaly'].sum()
normal_count = len(results) - outlier_count
total_count = len(results)
outlier_percent = (outlier_count / total_count) * 100

print("\n Statistik Outlier:")
print(f"Total Data     : {total_count}")
print(f"Normal Data    : {normal_count}")
print(f"Outlier Data   : {outlier_count}")
print(f"Persentase Outlier : {outlier_percent:.2f}%")

# Visualisasi tabel ringkas (opsional, misalnya untuk notebook)
summary_df = pd.DataFrame({
    'Kategori': ['Normal', 'Outlier'],
    'Jumlah': [normal_count, outlier_count],
    'Persentase': [100 - outlier_percent, outlier_percent]
})
print("\n Ringkasan Jumlah Data:")
print(summary_df)


# Buat scatter plot dengan dua fitur utama
plt.figure(figsize=(8,6))
plt.scatter(
    results['petal length'], results['petal width'],
    c=results['Anomaly'], cmap='coolwarm', edgecolor='k'
)

# Tambahkan judul dan label
plt.title("Visualisasi Deteksi Outlier dengan KNN")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

# Tambahkan legenda
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Normal',
               markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Outlier',
               markerfacecolor='red', markersize=8)
]
plt.legend(handles=handles)

plt.show()

```

#### Ringkasan Hasil Outlier Detection dengan Metode ABOD (10 Baris Pertama)

| Index | Sepal Length | Sepal Width | Petal Length | Petal Width | Anomaly | Anomaly Score |
|-------|--------------|-------------|--------------|-------------|---------|---------------|
| 131   | 7.9          | 3.8         | 6.4          | 2.0         | 1       | 1.024695      |
| 57    | 4.9          | 2.4         | 3.3          | 1.0         | 1       | 0.787401      |
| 109   | 7.2          | 3.6         | 6.1          | 2.5         | 1       | 0.806226      |
| 106   | 4.9          | 2.5         | 4.5          | 1.7         | 1       | 0.883176      |
| 117   | 7.7          | 3.8         | 6.7          | 2.2         | 1       | 1.019804      |
| 118   | 7.7          | 2.6         | 6.9          | 2.3         | 1       | 0.964365      |
| 98    | 5.1          | 2.5         | 3.0          | 1.1         | 1       | 0.818535      |
| 41    | 4.5          | 2.3         | 1.3          | 0.3         | 1       | 0.793726      |
| 84    | 5.4          | 3.0         | 4.5          | 1.5         | 0       | 0.509902      |
| 82    | 5.8          | 2.7         | 3.9          | 1.2         | 0       | 0.346410      |

---

#### Statistik Outlier KNN

- **Total Data**     : 150  
- **Normal Data**    : 142  
- **Outlier Data**   : 8  
- **Persentase Outlier** : 5.33%  

---

#### Ringkasan Jumlah Data KNN

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Normal   | 142    | 94.67%     |
| Outlier  | 8      | 5.33%      |

---

#### Visualisasi Outlier KNN


![Petal Length](knn.png)

---

### LOF

```python

import pandas as pd
from pycaret.anomaly import * 
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")
data = data.drop(columns=["iris"])

s = setup(data)

lof = create_model('lof') 
results = assign_model(lof)

results_sorted = results.sort_values(by="Anomaly", ascending=False)

# Ringkasan hasil outlier detection (tampilkan 10 baris pertama)
print("Ringkasan Hasil Outlier Detection (10 baris pertama):")
print(results_sorted[['sepal length','sepal width','petal length','petal width','Anomaly','Anomaly_Score']].head(10))

# Hitung jumlah outlier dan normal
outlier_count = results['Anomaly'].sum()
normal_count = len(results) - outlier_count
total_count = len(results)
outlier_percent = (outlier_count / total_count) * 100

print("\n Statistik Outlier:")
print(f"Total Data     : {total_count}")
print(f"Normal Data    : {normal_count}")
print(f"Outlier Data   : {outlier_count}")
print(f"Persentase Outlier : {outlier_percent:.2f}%")

# Visualisasi tabel ringkas (opsional, misalnya untuk notebook)
summary_df = pd.DataFrame({
    'Kategori': ['Normal', 'Outlier'],
    'Jumlah': [normal_count, outlier_count],
    'Persentase': [100 - outlier_percent, outlier_percent]
})
print("\n Ringkasan Jumlah Data:")
print(summary_df)


# Buat scatter plot dengan dua fitur utama
plt.figure(figsize=(8,6))
plt.scatter(
    results['sepal length'], results['petal width'],
    c=results['Anomaly'], cmap='coolwarm', edgecolor='k'
)

# Tambahkan judul dan label
plt.title("Visualisasi Deteksi Outlier dengan LOF")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Width")

# Tambahkan legenda
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Normal',
               markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Outlier',
               markerfacecolor='red', markersize=8)
]
plt.legend(handles=handles)

plt.show()

```

#### Ringkasan Hasil Outlier Detection dengan Metode LOF (10 Baris Pertama)

| Index | Sepal Length | Sepal Width | Petal Length | Petal Width | Anomaly | Anomaly Score |
|-------|--------------|-------------|--------------|-------------|---------|---------------|
| 122   | 7.7          | 2.8         | 6.7          | 2.0         | 1       | 1.491365      |
| 118   | 7.7          | 2.6         | 6.9          | 2.3         | 1       | 1.624653      |
| 41    | 4.5          | 2.3         | 1.3          | 0.3         | 1       | 1.777233      |
| 98    | 5.1          | 2.5         | 3.0          | 1.1         | 1       | 1.511868      |
| 15    | 5.7          | 4.4         | 1.5          | 0.4         | 1       | 1.653626      |
| 14    | 5.8          | 4.0         | 1.2          | 0.2         | 1       | 1.463125      |
| 131   | 7.9          | 3.8         | 6.4          | 2.0         | 1       | 1.547288      |
| 117   | 7.7          | 3.8         | 6.7          | 2.2         | 1       | 1.572990      |
| 106   | 4.9          | 2.5         | 4.5          | 1.7         | 0       | 1.374999      |
| 105   | 7.6          | 3.0         | 6.6          | 2.1         | 0       | 1.399554      |

---

#### Statistik Outlier LOF

- **Total Data**     : 150  
- **Normal Data**    : 142  
- **Outlier Data**   : 8  
- **Persentase Outlier** : 5.33%  

---

#### Ringkasan Jumlah Data LOF

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Normal   | 142    | 94.67%     |
| Outlier  | 8      | 5.33%      |

---

#### Visualisasi Outlier LOF


![Petal Length](lof.png)

---

## Kualitas Data
- **Konsistensi Data** 
  Semua kolom sudah memiliki format konsisten: numerik untuk ukuran bunga dan string untuk label spesies.  

- **Missing Value**  
  Tidak ditemukan nilai kosong (*null*) pada dataset.  

- **Duplikasi Data**  
  Setiap baris merepresentasikan sampel bunga unik, tidak ada duplikat.    

- **Balance Class**  
  Tiap kelas spesies memiliki jumlah data yang sama (50), sehingga tidak terjadi *class imbalance*.  

---

