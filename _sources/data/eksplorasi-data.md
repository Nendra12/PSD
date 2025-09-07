# Eksplorasi Data

Dataset yang digunakan adalah **Iris Dataset**, sebuah dataset klasik yang sering digunakan untuk pembelajaran machine learning dan analisis data. Dataset ini tersimpan dalam format **CSV** dengan jumlah **150 baris** (record) dan **5 kolom** (atribut). 

## Struktur Data  
- **Jumlah baris (record):** 150  
- **Jumlah kolom (atribut):** 5  
- **Tipe data:**  
  - **4 kolom numerik (kuantitatif):**  
    - *sepal length* (panjang sepal, dalam cm)  
    - *sepal width* (lebar sepal, dalam cm)  
    - *petal length* (panjang petal, dalam cm)  
    - *petal width* (lebar petal, dalam cm)  
  - **1 kolom kategorikal (kualitatif):**  
    - *iris* → menunjukkan spesies bunga dengan 3 kategori: **Iris-setosa, Iris-versicolor, dan Iris-virginica**  

## Contoh Data  
Berikut adalah 10 baris pertama dari dataset:  

![Table](table.jpg)

## Penyimpanan Data
Data iris yang berupa file csv saya simpan ke dalam cloud database (MySQL/PostgreSQL) agar bisa diakses di-query, atau diolah dengan lebih efisien.

**MySQL :**

![MySQL](mysql.jpg)

**PostgreSQL :**

![PostgreSQL](postgresql.jpg)

## Deskripsi Attribute pada Data

Data pada iris dataset terdiri dari beberapa kolom dengan tipe data berbeda. Kolom **iris** bertipe kualitatif yang berarti berisi data kategorikal (seperti nama spesies bunga: Iris-setosa, Iris-versicolor, atau Iris-virginica). Sedangkan itu **kolom sepal length, sepal width, petal length, dan petal width** bertipe angka Decimal, sehingga termasuk data numerik kontinu (kuantitatif) karena nilainya berupa bilangan pecahan yang bisa diukur dalam skala panjang.

## Statistik Deskriptif

![Statistik](hasil.jpg)

- **Petal Length:**  
  - Rata-rata: 3.76 cm
  - Min : 1.00 cm  
  - Max : 6.90 cm
- **Petal Width:**  
  - Rata-rata: 1.20 cm  
  - Min : 0.10 cm  
  - Max : 2.50 cm
- **Sepal Length:**  
  - Rata-rata: 5.84 cm  
  - Min : 4.30 cm  
  - Max : 7.90 cm
- **Sepal Width:**  
  - Rata-rata: 3.05 cm  
  - Min : 2.00 cm  
  - Max : 4.40 cm 

**Distribusi Kategori**  

Berdasarakan grafik batang, terlihat bahwa kolom kategori ini sama persis jumlahnya  
- Iris-setosa: 50 sampel  
- Iris-versicolor: 50 sampel  
- Iris-virginica: 50 sampel



