# Laporan Proyek Machine Learning - FATHUL KHOZIN

## Project Overview

Domain fashion memiliki karakteristik unik dibandingkan kategori produk lainnya. Preferensi pengguna terhadap pakaian sangat dipengaruhi oleh atribut visual dan deskriptif yang spesifik, seperti jenis pakaian (article type), warna dasar (base colour), penggunaan (usage), hingga gender. Seringkali, pengguna tidak mencari barang yang "populer" secara umum, melainkan barang yang memiliki karakteristik visual atau gaya yang mirip dengan barang yang sedang mereka lihat atau sukai sebelumnya. Oleh karena itu, pendekatan pencarian konvensional berbasis kata kunci sering kali tidak cukup untuk memenuhi kebutuhan eksplorasi gaya pengguna.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah:
- Keterbatasan Pengguna dalam Menelusuri Katalog Besar: Dengan ribuan produk fashion yang tersedia (lebih dari 40.000 item dalam dataset), pengguna mengalami kesulitan (pain point) untuk menemukan produk yang sesuai dengan preferensi visual mereka secara manual.
- Kebutuhan akan Rekomendasi Serupa: Pengguna sering kali menyukai gaya atau karakteristik tertentu dari suatu produk (misalnya warna atau kategori) tetapi mungkin ingin melihat variasi lain yang mirip. Bagaimana cara sistem dapat mengidentifikasi dan menyarankan produk lain yang memiliki kemiripan karakteristik tersebut secara otomatis?
### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan Sistem Rekomendasi Berbasis Konten: Membuat model machine learning yang mampu menganalisis fitur-fitur teks pada produk fashion (seperti deskripsi, warna, dan kategori) untuk memahami karakteristik setiap item.
- Menghasilkan Top-N Recommendation: Memberikan daftar rekomendasi produk (Top-N) yang memiliki tingkat kemiripan tertinggi dengan produk yang sedang dilihat atau disukai oleh pengguna, guna membantu pengguna menemukan produk relevan dengan lebih cepat.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
    ### Solution statements
    - Untuk mencapai tujuan tersebut, pendekatan solusi yang akan digunakan adalah Content-based Filtering. Sesuai dengan karakteristik masalah di mana rekomendasi didasarkan pada atribut item, pendekatan ini dipilih dengan alur sebagai berikut: Content-based Filtering: Ide utama dari pendekatan ini adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu atau sedang dilihat saat ini. Kesamaan (similarity) antar item dihitung berdasarkan fitur-fitur yang melekat pada item tersebut. Teknik yang digunakan:
  1.   TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency): Digunakan untuk mengekstraksi fitur dari data teks (gabungan kategori, warna, dan nama produk). Teknik ini akan memberikan bobot pada kata-kata penting yang menjadi ciri khas suatu produk.
  2.   Cosine Similarity: Digunakan untuk menghitung derajat kesamaan (jarak) antar produk berdasarkan representasi vektor yang dihasilkan oleh TF-IDF. Semakin kecil sudut antar vektor, semakin mirip kedua produk tersebut.


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Fashion Recommendation Dataset yang diunduh dari platform Kaggle. Dataset ini berisi metadata produk fashion yang mencakup berbagai kategori seperti pakaian, sepatu, dan aksesoris.
- Sumber Data: Fashion Recommendation Dataset - Kaggle (https://www.kaggle.com/datasets/adityakammati/fashion-recomendation-dataset/data)
- Format Data: CSV (Comma-Separated Values).
- Jumlah Data: Dataset memiliki total 44.419 baris (records).

Kondisi Data: Setiap baris merepresentasikan satu item produk unik yang dilengkapi dengan atribut-atribut spesifik. Terdapat beberapa missing values pada kolom deskriptif tertentu yang akan ditangani pada tahap Data Preparation.

Variabel-variabel pada Fashion Recommendation Dataset - Kaggle dataset adalah sebagai berikut:
- id: Identitas unik untuk setiap produk (Integer).
- gender: Target pengguna produk (kategori: Men, Women, Boys, Girls, Unisex).
- masterCategory: Kategori utama produk (contoh: Apparel, Accessories, Footwear).
- subCategory: Sub-kategori yang lebih spesifik (contoh: Topwear, Shoes, Bags).
- articleType: Jenis produk yang sangat spesifik (contoh: T-shirt, Watches, Casual Shoes).
- baseColour: Warna dominan dari produk.
- usage: Peruntukan penggunaan produk (contoh: Casual, Sports, Ethnic).
- productDisplayName: Nama atau judul tampilan produk yang berisi deskripsi singkat.
**Rubrik/Kriteria Tambahan (Opsional)**:
- Distribusi Produk Berdasarkan Master Category.
	Berdasarkan visualisasi di atas, terlihat bahwa mayoritas produk dalam dataset didominasi oleh kategori Apparel (Pakaian), diikuti oleh Accessories dan Footwear. Hal ini menunjukkan bahwa sistem rekomendasi yang dibangun akan memiliki variasi data yang sangat kaya terutama untuk rekomendasi pakaian.
- Distribusi Produk Berdasarkan Gender.
	Data menunjukkan persebaran yang cukup berimbang antara produk untuk Men (Pria) dan Women (Wanita), dengan sejumlah kecil produk untuk kategori anak-anak (Boys/Girls) dan Unisex. Ini mengindikasikan bahwa model dapat digunakan secara efektif untuk merekomendasikan produk bagi pengguna pria maupun wanita.
- Distribusi Warna (Top 10).
	Warna-warna netral seperti Black, White, dan Blue merupakan warna yang paling dominan dalam koleksi produk. Informasi ini penting karena warna adalah salah satu atribut visual utama dalam menentukan kemiripan produk pada Content-Based Filtering.

## Data Preparation
1. Feature Selection (Seleksi Fitur)
Dataset awal memiliki banyak kolom informasi, namun tidak semuanya relevan untuk menentukan kemiripan visual antar produk. Oleh karena itu, dilakukan seleksi fitur untuk mengambil atribut-atribut kunci yang merepresentasikan karakteristik produk.
- Fitur yang dipilih: gender, masterCategory, subCategory, articleType, baseColour, dan productDisplayName.
- Alasan: Atribut-atribut ini mengandung informasi deskriptif yang paling kuat untuk membedakan satu produk fashion dengan yang lainnya (misalnya membedakan "Kemeja Pria Merah" dengan "Tas Wanita Biru")
2. Data Cleaning (Penanganan Missing Values)
Algoritma sistem rekomendasi tidak dapat memproses data yang kosong (null/NaN). Pada dataset ini, ditemukan beberapa baris yang memiliki nilai kosong pada kolom deskriptif.
- Tindakan: Mengisi nilai yang hilang (missing values) dengan string kosong ('').
- Alasan: Langkah ini dilakukan untuk mencegah error saat proses manipulasi string dan memastikan setiap produk memiliki representasi teks yang lengkap, meskipun ada atribut yang tidak tersedia.
3. Feature Engineering (Membuat "Tags" / Soup)
Agar algoritma Content-Based Filtering dapat menghitung kemiripan, seluruh atribut teks yang terpisah-pisah perlu disatukan menjadi satu kesatuan dokumen.
- Tindakan: Membuat kolom baru bernama tags yang merupakan gabungan dari kolom gender, masterCategory, subCategory, articleType, baseColour, dan productDisplayName.
- Contoh: Jika sebuah produk adalah baju pria berwarna biru, maka kolom tags akan berisi: "Men Apparel Topwear T-shirt Blue Nike Polo..."
Alasan: Penyatuan ini memudahkan proses vektorisasi (TF-IDF), di mana model akan menganggap gabungan teks tersebut sebagai "dokumen" tunggal yang memuat seluruh identitas produk.
4. Data Sampling (Opsional/Kondisional)
Mengingat besarnya jumlah data (44.000+) dan keterbatasan memori (RAM) pada lingkungan komputasi Google Colab saat menghitung matriks kesamaan (similarity matrix), dilakukan pengambilan sampel data.
- Tindakan: Menggunakan 10.000 data pertama untuk proses pemodelan.
- Alasan: Untuk efisiensi komputasi dan mencegah crash (Memory Error) tanpa mengurangi esensi dari pembuktian konsep (Proof of Concept) sistem rekomendasi ini.
## Modeling
Modeling and Result
Pada tahap ini, dilakukan pembangunan model sistem rekomendasi untuk menjawab permasalahan yang telah didefinisikan sebelumnya. Model ini bertujuan untuk memberikan rekomendasi produk fashion yang relevan berdasarkan kemiripan konten atau fitur atribut produk
1. Algoritma: Content-Based Filtering
Proyek ini menggunakan algoritma Content-Based Filtering. Prinsip utama algoritma ini adalah memberikan rekomendasi kepada pengguna berdasarkan kemiripan antara item yang disukai di masa lalu dengan item baru di dalam katalog. Langkah-langkah pemodelan yang dilakukan adalah sebagai berikut:
Vektorisasi dengan TF-IDF Vectorizer: Teknik TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah data tekstual pada kolom tags (gabungan fitur produk) menjadi representasi angka berupa matriks. Teknik ini menghitung frekuensi kemunculan kata dalam suatu item dan memberikan bobot lebih tinggi pada kata-kata yang bersifat unik atau spesifik sebagai ciri khas produk tersebut.
Perhitungan Similarity dengan Cosine Similarity: Setelah data produk direpresentasikan dalam bentuk vektor angka, derajat kesamaan antar produk dihitung menggunakan Cosine Similarity. Teknik ini mengukur sudut antara dua vektor di dalam ruang multidimensi. Semakin kecil sudutnya (skor mendekati 1.0), maka semakin tinggi tingkat kemiripan antar kedua produk tersebut.
2. Output: Top-N Recommendation
Sistem ini dirancang untuk menghasilkan Top-10 Recommendation. Ketika pengguna memasukkan satu judul produk sebagai input, sistem akan mencari 10 produk lain yang memiliki skor Cosine Similarity tertinggi.
Berikut adalah contoh hasil rekomendasi yang dihasilkan oleh model:
Produk Input: Turtle Check Men Navy Blue Shirt
NoProduct Display NameMaster CategoryArticle Type
  1.   Turtle Men Check Blue ShirtApparelShirts
  2.   Turtle Men Check Black ShirtApparelShirts
  3.   Turtle Men Striped Blue ShirtApparelShirts
  4.   Peter England Men Check Blue ShirtApparelShirts
Analisis Hasil: Berdasarkan output di atas, sistem berhasil memberikan rekomendasi yang sangat relevan. Untuk input berupa "Kemeja Kotak-kotak Biru", sistem mampu menyodorkan produk lain yang juga merupakan kemeja dengan motif atau warna serupa dari berbagai merek. Hal ini menunjukkan bahwa fitur-fitur yang digabungkan dalam tahap Data Preparation berhasil diekstraksi dengan baik oleh model.
3. Kelebihan dan Kekurangan Pendekatan
Pemilihan pendekatan Content-Based Filtering memiliki beberapa konsekuensi sebagai berikut:
Kelebihan:
No Cold-Start for Items: Sistem dapat merekomendasikan item baru meskipun belum pernah ada pengguna yang berinteraksi atau membelinya, asalkan deskripsi fiturnya lengkap.
User Independence: Rekomendasi bersifat personal dan hanya bergantung pada profil atau item yang dilihat pengguna saat ini, tanpa memerlukan data dari pengguna lain.
Kekurangan:
Overspecialization: Sistem cenderung merekomendasikan item yang "terlalu mirip" dengan apa yang sudah diketahui pengguna, sehingga pengguna jarang mendapatkan rekomendasi item yang berbeda jenis namun mungkin disukai (serendipity).
Keterbatasan Fitur: Akurasi sistem sangat bergantung pada seberapa detail dan akurat informasi teks (metadata) yang tersedia pada dataset.

## Evaluation
1. Metrik Evaluasi: Precision@K
Metrik yang digunakan pada sistem ini adalah Precision at K (Presisi pada K rekomendasi teratas).
Definisi: Precision@K mengukur proporsi rekomendasi yang relevan di dalam daftar Top-K rekomendasi yang dihasilkan oleh sistem.
Definisi Relevansi: Dalam konteks proyek ini, sebuah rekomendasi dianggap relevan jika produk yang direkomendasikan memiliki subCategory (Sub-kategori) yang sama dengan produk input.
Contoh: Jika pengguna mencari "Sepatu Lari" (Shoes), maka rekomendasi dianggap relevan jika yang muncul adalah produk-produk lain dalam kategori "Sepatu" (Shoes), bukan "Baju" (Topwear).
2. Formula Metrik
Formula untuk menghitung Precision@K adalah sebagai berikut:
Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.
Precision@K = Jumlah Item Relevan dalam Top-K/K

Dimana:
Jumlah Item Relevan: Banyaknya produk rekomendasi yang memiliki kategori sama dengan produk input.
K: Jumlah total rekomendasi yang diberikan (pada proyek ini K=10).
3. Hasil Evaluasi
Untuk menguji performa model, dilakukan pengujian sampel terhadap produk dengan kategori yang berbeda. Berikut adalah hasil perhitungan Precision@10:
Sampel Pengujian:
Input: Puma Men Grey T-shirt (Kategori: Topwear)
Hasil Rekomendasi: 10 produk Topwear (Kaos/Kemeja).
Precision: 10/10 = 100%
Input: Sonata Women Gold Watch (Kategori: Watches)
Hasil Rekomendasi: 9 Jam tangan, 1 Gelang.
Precision: 10/10 = 90%
Rata-rata Presisi: Berdasarkan pengujian terhadap beberapa sampel acak dari berbagai kategori, model sistem rekomendasi ini memiliki rata-rata presisi di atas 90%. Hal ini menunjukkan bahwa pendekatan TF-IDF dan Cosine Similarity sangat efektif dalam mengelompokkan produk yang serupa berdasarkan fitur teksnya.
