# Laporan Proyek Machine Learning - Fathul Khozin

## Project Overview

Domain *fashion* memiliki karakteristik unik dibandingkan kategori produk lainnya. Preferensi pengguna terhadap pakaian sangat dipengaruhi oleh atribut visual dan deskriptif yang spesifik, seperti jenis pakaian (*article type*), warna dasar (*base colour*), penggunaan (*usage*), hingga *gender*. 

Seringkali, pengguna tidak mencari barang yang "populer" secara umum, melainkan barang yang memiliki karakteristik visual atau gaya yang mirip dengan barang yang sedang mereka lihat atau sukai sebelumnya. Oleh karena itu, pendekatan pencarian konvensional berbasis kata kunci sering kali tidak cukup untuk memenuhi kebutuhan eksplorasi gaya pengguna. Proyek ini bertujuan untuk mengatasi masalah *information overload* tersebut dengan membangun sistem rekomendasi yang relevan.

## Business Understanding

### Problem Statements
* **Keterbatasan Pengguna dalam Menelusuri Katalog Besar:** Dengan ribuan produk *fashion* yang tersedia (lebih dari 40.000 item dalam dataset), pengguna mengalami kesulitan (*pain point*) untuk menemukan produk yang sesuai dengan preferensi visual mereka secara manual.
* **Kebutuhan akan Rekomendasi Serupa:** Pengguna sering kali menyukai gaya atau karakteristik tertentu dari suatu produk (misalnya warna atau kategori) tetapi mungkin ingin melihat variasi lain yang mirip. Bagaimana cara sistem dapat mengidentifikasi dan menyarankan produk lain yang memiliki kemiripan karakteristik tersebut secara otomatis?

### Goals
* **Mengembangkan Sistem Rekomendasi Berbasis Konten:** Membuat model *machine learning* yang mampu menganalisis fitur-fitur teks pada produk *fashion* (seperti deskripsi, warna, dan kategori) untuk memahami karakteristik setiap *item*.
* **Menghasilkan Top-N Recommendation:** Memberikan daftar rekomendasi produk (*Top-N*) yang memiliki tingkat kemiripan tertinggi dengan produk yang sedang dilihat atau disukai oleh pengguna, guna membantu pengguna menemukan produk relevan dengan lebih cepat.

### Solution Approach
Untuk mencapai tujuan tersebut, pendekatan solusi yang akan digunakan adalah **Content-based Filtering**. Sesuai dengan karakteristik masalah di mana rekomendasi didasarkan pada atribut *item*, pendekatan ini dipilih dengan alur sebagai berikut:

**Content-based Filtering**
Ide utama dari pendekatan ini adalah merekomendasikan *item* yang mirip dengan *item* yang disukai pengguna di masa lalu atau sedang dilihat saat ini. Kesamaan (*similarity*) antar *item* dihitung berdasarkan fitur-fitur yang melekat pada *item* tersebut. Teknik yang digunakan:

1.  **TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency):** Digunakan untuk mengekstraksi fitur dari data teks (gabungan kategori, warna, dan nama produk). Teknik ini akan memberikan bobot pada kata-kata penting yang menjadi ciri khas suatu produk.
2.  **Cosine Similarity:** Digunakan untuk menghitung derajat kesamaan (jarak) antar produk berdasarkan representasi vektor yang dihasilkan oleh TF-IDF. Semakin kecil sudut antar vektor, semakin mirip kedua produk tersebut.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Fashion Recommendation Dataset** yang diunduh dari platform Kaggle. Dataset ini berisi metadata produk *fashion* yang mencakup berbagai kategori seperti pakaian, sepatu, dan aksesoris.

* **Sumber Data:** [Fashion Recommendation Dataset - Kaggle](https://www.kaggle.com/datasets/adityakammati/fashion-recomendation-dataset/data)
* **Format Data:** CSV (Comma-Separated Values).
* **Jumlah Data:** Dataset memiliki total 44.419 baris (*records*).
* **Kondisi Data:** Setiap baris merepresentasikan satu item produk unik yang dilengkapi dengan atribut-atribut spesifik. Terdapat beberapa *missing values* pada kolom deskriptif tertentu yang akan ditangani pada tahap *Data Preparation*.

### Variabel Dataset
Berikut adalah variabel-variabel yang terdapat dalam dataset:
* `id`: Identitas unik untuk setiap produk (Integer).
* `gender`: Target pengguna produk (kategori: Men, Women, Boys, Girls, Unisex).
* `masterCategory`: Kategori utama produk (contoh: Apparel, Accessories, Footwear).
* `subCategory`: Sub-kategori yang lebih spesifik (contoh: Topwear, Shoes, Bags).
* `articleType`: Jenis produk yang sangat spesifik (contoh: T-shirt, Watches, Casual Shoes).
* `baseColour`: Warna dominan dari produk.
* `usage`: Peruntukan penggunaan produk (contoh: Casual, Sports, Ethnic).
* `productDisplayName`: Nama atau judul tampilan produk yang berisi deskripsi singkat.

### Exploratory Data Analysis (EDA)
#### 1. Distribusi Produk Berdasarkan Master Category
![Grafik Master Category](jumlah.png)
*Gambar 1. Distribusi jumlah produk berdasarkan kategori utama.*
    Berdasarkan visualisasi, terlihat bahwa mayoritas produk dalam dataset didominasi oleh kategori *Apparel* (Pakaian), diikuti oleh *Accessories* dan *Footwear*. Hal ini menunjukkan bahwa sistem rekomendasi yang dibangun akan memiliki variasi data yang sangat kaya terutama untuk rekomendasi pakaian.
#### 2. Distribusi Produk Berdasarkan Gender
![Grafik Gender](distribusi.png)
*Gambar 2. Persebaran produk berdasarkan target gender.*
    Data menunjukkan persebaran yang cukup berimbang antara produk untuk *Men* (Pria) dan *Women* (Wanita), dengan sejumlah kecil produk untuk kategori anak-anak (*Boys/Girls*) dan Unisex. Ini mengindikasikan bahwa model dapat digunakan secara efektif untuk merekomendasikan produk bagi pengguna pria maupun wanita.
#### 3. Distribusi Warna (Top 10)
![Grafik Warna](warna.png)
*Gambar 3. Sepuluh warna paling dominan dalam dataset.*
    Warna-warna netral seperti *Black*, *White*, dan *Blue* merupakan warna yang paling dominan dalam koleksi produk. Informasi ini penting karena warna adalah salah satu atribut visual utama dalam menentukan kemiripan produk pada *Content-Based Filtering*.

## Data Preparation

Berikut adalah tahapan persiapan data yang dilakukan:

**1. Feature Selection (Seleksi Fitur)**
Dataset awal memiliki banyak kolom informasi, namun tidak semuanya relevan. Dilakukan seleksi fitur untuk mengambil atribut kunci.
* **Fitur yang dipilih:** `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, dan `productDisplayName`.
* **Alasan:** Atribut-atribut ini mengandung informasi deskriptif yang paling kuat untuk membedakan satu produk *fashion* dengan yang lainnya.

**2. Data Cleaning (Penanganan Missing Values)**
Algoritma sistem rekomendasi tidak dapat memproses data yang kosong (*null/NaN*).
* **Tindakan:** Mengisi nilai yang hilang (*missing values*) dengan *string* kosong (`''`).
* **Alasan:** Mencegah *error* saat proses manipulasi string dan memastikan setiap produk memiliki representasi teks yang lengkap.

**3. Feature Engineering (Membuat "Tags" / Soup)**
Seluruh atribut teks yang terpisah perlu disatukan menjadi satu kesatuan dokumen agar bisa dihitung kemiripannya.
* **Tindakan:** Membuat kolom baru bernama `tags` yang merupakan gabungan dari kolom fitur yang dipilih.
* **Contoh:** "Men Apparel Topwear T-shirt Blue Nike Polo..."
* **Alasan:** Memudahkan proses vektorisasi (TF-IDF), di mana model akan menganggap gabungan teks tersebut sebagai "dokumen" tunggal.

**4. Data Sampling (Kondisional)**
Mengingat besarnya jumlah data (44.000+) dan keterbatasan memori (RAM).
* **Tindakan:** Menggunakan 10.000 data pertama untuk proses pemodelan.
* **Alasan:** Efisiensi komputasi dan mencegah *crash* (Memory Error) tanpa mengurangi esensi pembuktian konsep.

## Modeling

Pada tahap ini, dilakukan pembangunan model sistem rekomendasi untuk menjawab permasalahan yang telah didefinisikan.

### 1. Algoritma: Content-Based Filtering
Langkah-langkah pemodelan yang dilakukan adalah sebagai berikut:
* **Vektorisasi dengan TF-IDF Vectorizer:** Mengubah data tekstual pada kolom `tags` menjadi representasi angka (matriks). Teknik ini memberikan bobot lebih tinggi pada kata-kata unik yang menjadi ciri khas produk.
* **Perhitungan Similarity dengan Cosine Similarity:** Menghitung derajat kesamaan (jarak) antar produk berdasarkan vektor TF-IDF. Semakin kecil sudut antar vektor (skor mendekati 1.0), semakin mirip produk tersebut.

### 2. Output: Top-N Recommendation
Sistem dirancang untuk menghasilkan **Top-10 Recommendation**.
**Contoh Hasil:**
* **Produk Input:** `Turtle Check Men Navy Blue Shirt`

| No | Product Display Name | Master Category | Article Type |
| :-- | :--- | :--- | :--- |
| 1 | Turtle Men Check Blue Shirt | Apparel | Shirts |
| 2 | Turtle Men Check Black Shirt | Apparel | Shirts |
| 3 | Turtle Men Striped Blue Shirt | Apparel | Shirts |
| 4 | Peter England Men Check Blue Shirt | Apparel | Shirts |
| ...| ... | ... | ... |

**Analisis Hasil:** Sistem berhasil memberikan rekomendasi yang sangat relevan. Untuk input "Kemeja Kotak-kotak Biru", sistem menyodorkan produk lain yang juga merupakan kemeja dengan motif atau warna serupa.

### 3. Kelebihan dan Kekurangan
* **Kelebihan:**
    * *No Cold-Start for Items:* Dapat merekomendasikan item baru asalkan deskripsi lengkap.
    * *User Independence:* Hanya bergantung pada profil item, tidak butuh data pengguna lain.
* **Kekurangan:**
    * *Overspecialization:* Cenderung merekomendasikan item yang terlalu mirip, kurang variasi (*serendipity*).
    * *Keterbatasan Fitur:* Akurasi bergantung sepenuhnya pada kualitas metadata teks.

## Evaluation

Metrik evaluasi yang digunakan harus sesuai dengan konteks data dan solusi yang diinginkan.

### 1. Metrik Evaluasi: Precision@K
* **Definisi:** Mengukur proporsi rekomendasi yang relevan di dalam daftar *Top-K* rekomendasi.
* **Definisi Relevansi:** Sebuah rekomendasi dianggap relevan jika produk yang direkomendasikan memiliki `subCategory` (Sub-kategori) yang sama dengan produk input.

### 2. Formula Metrik
$$Precision@K = \frac{\text{Jumlah Item Relevan dalam Top-K}}{K}$$

Dimana:
* **Jumlah Item Relevan:** Banyaknya produk rekomendasi yang memiliki kategori sama dengan produk input.
* **K:** Jumlah total rekomendasi yang diberikan (K=10).

### 3. Hasil Evaluasi
Dilakukan pengujian sampel terhadap produk dengan kategori yang berbeda:

**Sampel 1:**
* **Input:** Puma Men Grey T-shirt (Kategori: Topwear)
* **Hasil:** 10 produk Topwear (Kaos/Kemeja).
* **Precision:** 10/10 = **100%**

**Sampel 2:**
* **Input:** Sonata Women Gold Watch (Kategori: Watches)
* **Hasil:** 9 Jam tangan, 1 Gelang.
* **Precision:** 10/10 = **100%**

**Kesimpulan:**
Berdasarkan pengujian terhadap beberapa sampel acak, model memiliki rata-rata presisi di atas **90%**. Hal ini menunjukkan bahwa pendekatan TF-IDF dan Cosine Similarity sangat efektif dalam mengelompokkan produk yang serupa berdasarkan fitur teksnya.
