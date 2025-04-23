# **Laporan Proyek Machine Learning - Hafiyan Al Muqaffi Umary**

## **Domain Proyek**

Investasi properti merupakan salah satu bentuk investasi yang banyak diminati oleh masyarakat. Namun, menentukan harga properti yang tepat masih menjadi tantangan tersendiri karena dipengaruhi oleh banyak faktor seperti lokasi, ukuran, fasilitas, dan kondisi properti. Ketidaktepatan dalam estimasi harga dapat menyebabkan kerugian finansial yang signifikan bagi penjual maupun pembeli.

Prediksi harga rumah telah menjadi salah satu aplikasi yang penting dalam bidang data science dan machine learning. Dengan memanfaatkan data historis penjualan properti, model prediktif dapat membantu memperkirakan harga properti berdasarkan berbagai fitur yang mempengaruhinya. Hal ini memungkinkan agen real estate, developer, dan pembeli untuk membuat keputusan yang lebih informed dalam transaksi properti [1].

Menurut penelitian yang dilakukan oleh Case dan Shiller [2], harga properti memiliki pola tertentu dan dipengaruhi oleh berbagai faktor ekonomi dan karakteristik fisik dari properti itu sendiri. Penelitian lain oleh Limsombunchai [3] menunjukkan bahwa teknik machine learning seperti regresi dan neural network dapat secara efektif memprediksi harga properti dengan tingkat kesalahan yang dapat diterima.

Pendekatan machine learning untuk prediksi harga rumah dapat memberikan manfaat ekonomi yang signifikan. Menurut Peterson dan Flanagan [4], model prediksi yang akurat dapat meningkatkan efisiensi pasar perumahan dengan mengurangi asimetri informasi antara penjual dan pembeli. Selain itu, Jenis model ini juga dapat membantu meningkatkan transparansi dan likuiditas pasar properti.

[1] [Housing Price Prediction Using Machine Learning Algorithms](https://www.sciencedirect.com/science/article/pii/S1877050920304865)  
[2] [Case, K. E., & Shiller, R. J. (1989). The Efficiency of the Market for Single-Family Homes](https://scholar.google.com/scholar?q=Case+Shiller+The+Efficiency+of+the+Market+for+Single+Family+Homes)  
[3] [Limsombunchai, V. (2004). House Price Prediction: Hedonic Price Model vs. Artificial Neural Network](https://scholar.google.com/scholar?q=Limsombunchai+House+Price+Prediction+Hedonic+Price+Model+vs+Artificial+Neural+Network)  
[4] [Peterson, S., & Flanagan, A. (2009). Neural Network Hedonic Pricing Models in Mass Real Estate Appraisal](https://scholar.google.com/scholar?q=Peterson+Flanagan+Neural+Network+Hedonic+Pricing+Models+in+Mass+Real+Estate+Appraisal)

## **Business Understanding**

### **Problem Statements**

Beberapa permasalahan yang akan diselesaikan dalam proyek ini:

* Bagaimana mengembangkan model machine learning yang dapat memprediksi harga rumah dengan akurat berdasarkan fitur-fitur yang tersedia?  
* Faktor-faktor atau fitur apa saja yang paling berpengaruh terhadap harga rumah?  
* Seberapa akurat model machine learning dapat memprediksi harga rumah dibandingkan dengan harga aktual?

### **Goals**

Tujuan dari proyek ini adalah:

* Mengembangkan model prediksi harga rumah yang memiliki tingkat error rendah dengan menggunakan algoritma machine learning.  
* Mengidentifikasi fitur-fitur yang memiliki korelasi dan pengaruh paling signifikan terhadap harga rumah.  
* Menghasilkan model prediksi yang memiliki error (RMSE) kurang dari 20% dari rata-rata harga rumah dalam dataset.

### **Solution Statements**

Untuk mencapai tujuan yang telah ditetapkan, beberapa solusi yang akan diimplementasikan:

1. Menggunakan algoritma Linear Regression sebagai baseline model untuk memprediksi harga rumah karena algoritma ini sederhana dan interpretable, sehingga cocok untuk memahami pengaruh setiap fitur terhadap target.

2. Mengimplementasikan algoritma Decision Tree Regressor yang mampu menangkap hubungan non-linear antara fitur dan target.

3. Menerapkan algoritma Random Forest Regressor yang dapat menangani fitur dengan berbagai skala dan mampu menangkap interaksi kompleks antar fitur.

4. Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa prediksi.

5. Menggunakan metrik evaluasi seperti MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), dan R-squared untuk mengukur performa model.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini adalah House Price Prediction dataset yang tersedia di [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). Dataset ini berisi informasi penjualan rumah di King County, USA selama periode Mei 2014 hingga Mei 2015. Dataset terdiri dari 21.613 records dan 21 columns (fitur).

### **Variabel-variabel pada House Price Prediction dataset adalah sebagai berikut:**

1. id: Pengenal unik untuk setiap rumah  
2. date: Tanggal rumah terjual  
3. price: Harga rumah (target prediksi)  
4. bedrooms: Jumlah kamar tidur  
5. bathrooms: Jumlah kamar mandi  
6. sqft_living: Luas bangunan dalam satuan kaki persegi  
7. sqft_lot: Luas tanah dalam satuan kaki persegi  
8. floors: Jumlah lantai  
9. waterfront: Indikator apakah rumah menghadap ke perairan (0 = tidak, 1 = ya)  
10. view: Indeks dari 0 hingga 4 mengukur kualitas pemandangan dari properti  
11. condition: Indeks dari 1 hingga 5 mengukur kondisi rumah  
12. grade: Indeks dari 1 hingga 13 mengukur kualitas konstruksi dan desain bangunan  
13. sqft_above: Luas area di atas permukaan tanah dalam kaki persegi  
14. sqft_basement: Luas area basement dalam kaki persegi  
15. yr_built: Tahun rumah dibangun  
16. yr_renovated: Tahun terakhir rumah direnovasi  
17. zipcode: Kode pos lokasi rumah  
18. lat: Koordinat latitude  
19. long: Koordinat longitude  
20. sqft_living15: Luas area hunian pada tahun 2015 (dapat berbeda dari sqft_living jika ada renovasi)  
21. sqft_lot15: Luas area tanah pada tahun 2015 (dapat berbeda dari sqft_lot jika ada perubahan)

### **Exploratory Data Analysis**

Untuk memahami dataset dengan lebih baik, dilakukan analisis eksplorasi terhadap data. Berikut beberapa insight yang diperoleh:

#### **Struktur Data**

Dataset memiliki 21,613 baris dan 21 kolom. Dari hasil pengecekan tipe data, semua kolom memiliki tipe data yang sesuai dengan karakteristik informasi yang disimpan (numerik untuk fitur seperti price, bedrooms, dan object untuk tanggal).

```
Shape: (21613, 21)

Data Types:
id                 int64
date              object
price            float64
bedrooms           int64
bathrooms        float64
sqft_living        int64
sqft_lot           int64
floors           float64
waterfront         int64
view               int64
condition          int64
grade              int64
sqft_above         int64
sqft_basement      int64
yr_built           int64
yr_renovated       int64
zipcode            int64
lat              float64
long             float64
sqft_living15      int64
sqft_lot15         int64
dtype: object
```

#### **Statistik Deskriptif**

Statistik deskriptif menunjukkan bahwa rata-rata harga rumah adalah sekitar $540,088, dengan harga minimum $75,000 dan maksimum $7,700,000. Rata-rata rumah memiliki 3.37 kamar tidur dan 2.11 kamar mandi. Luas bangunan rata-rata adalah sekitar 2,080 kaki persegi.

#### **Distribusi Harga Rumah**

Analisis distribusi harga rumah menunjukkan bahwa sebagian besar rumah memiliki harga antara $300,000 hingga $700,000, dengan beberapa outlier yang memiliki harga sangat tinggi. Distribusi harga cenderung skewed to the right (condong ke kanan).

#### **Korelasi antar Fitur**

Analisis korelasi menunjukkan bahwa fitur sqft_living (luas bangunan) memiliki korelasi positif yang kuat dengan harga rumah. Fitur lain seperti grade, sqft_above, dan sqft_living15 juga menunjukkan korelasi positif yang signifikan dengan harga.

#### **Analisis Lokasi**

Analisis berdasarkan koordinat geografis (latitude dan longitude) menunjukkan bahwa lokasi sangat berpengaruh terhadap harga rumah. Rumah-rumah di dekat pusat kota dan area perairan cenderung memiliki harga yang lebih tinggi.

#### **Pengaruh Renovasi**

Rumah yang telah direnovasi umumnya memiliki harga yang lebih tinggi dibandingkan dengan rumah yang belum pernah direnovasi, menunjukkan bahwa renovasi dapat meningkatkan nilai properti.

## **Data Preparation**

Beberapa teknik data preparation yang diterapkan dalam proyek ini:

### **1. Handling Missing Values**

Meskipun dataset ini relatif bersih dan tidak terdapat missing values, kita tetap melakukan penanganan untuk berjaga-jaga pada beberapa kolom potensial:

- waterfront diisi dengan 0 (asumsi default tidak menghadap perairan)
- view diisi dengan 0 (asumsi default tidak ada view)
- yr_renovated diisi dengan 0 (asumsi default belum direnovasi)

### **2. Feature Engineering**

Beberapa fitur baru dibuat untuk meningkatkan performa model:

- age: usia rumah (2015 - yr_built)
- renovated: status renovasi (1 jika pernah direnovasi, 0 jika belum)
- total_area: total luas area (sqft_living + sqft_lot)
- price_per_sqft: harga per kaki persegi (price / sqft_living)
- sale_year, sale_month, sale_day: ekstraksi komponen tanggal dari kolom date

### **3. Handling Outliers**

Outliers pada harga rumah dan luas bangunan dapat mempengaruhi performa model, sehingga dilakukan penanganan outliers menggunakan metode IQR (Interquartile Range):

- Menghitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga) untuk kolom price
- Menghitung IQR = Q3 - Q1
- Menentukan batas bawah = Q1 - 1.5 * IQR
- Menentukan batas atas = Q3 + 1.5 * IQR
- Memfilter data yang berada di luar batas tersebut

### **4. Feature Scaling**

Beberapa algoritma machine learning sensitif terhadap skala fitur. Oleh karena itu, dilakukan normalisasi pada fitur-fitur numerik menggunakan StandardScaler agar memiliki skala yang sama dengan mean 0 dan standar deviasi 1.

### **5. Feature Selection**

Untuk mengurangi dimensionalitas dan fokus pada fitur-fitur yang paling relevan, dilakukan feature selection berdasarkan korelasi dengan target. Diambil 15 fitur teratas yang memiliki korelasi tertinggi dengan harga rumah.

### **6. One-Hot Encoding**

Untuk variabel kategorikal seperti zipcode, dilakukan one-hot encoding agar dapat diproses oleh algoritma machine learning. Transformasi ini mengubah variabel kategorikal menjadi beberapa kolom biner yang merepresentasikan keberadaan kategori tersebut.

## **Modeling**

Pada tahap ini, tiga algoritma machine learning diterapkan untuk memprediksi harga rumah: Linear Regression, Decision Tree Regressor, dan Random Forest Regressor.

### **1. Data Splitting**

Sebelum melakukan pemodelan, dataset dibagi menjadi data training (80%) dan data testing (20%) dengan random_state=42 untuk memastikan hasil yang konsisten dan dapat direproduksi.

### **2. Linear Regression**

Model Linear Regression diimplementasikan sebagai baseline model karena kesederhanaan dan interpretabilitasnya.

**Kelebihan Linear Regression:**
* Mudah diimplementasikan dan diinterpretasi
* Komputasi yang efisien dan cepat
* Memberikan insight tentang hubungan linear antara fitur dan target

**Kekurangan Linear Regression:**
* Hanya dapat menangkap hubungan linear
* Sensitif terhadap outliers
* Tidak dapat menangkap interaksi kompleks antar fitur

### **3. Decision Tree Regressor**

Model Decision Tree Regressor dapat menangkap hubungan non-linear dalam data.

**Parameter yang digunakan:**
- random_state=42 untuk reproducibility
- Parameter lain menggunakan nilai default

**Kelebihan Decision Tree Regressor:**
* Dapat menangkap hubungan non-linear
* Mudah divisualisasikan dan diinterpretasi
* Tidak memerlukan feature scaling

**Kekurangan Decision Tree Regressor:**
* Rentan terhadap overfitting
* Tidak stabil (small changes in data can lead to large changes in structure)
* Performanya seringkali tidak sebaik model ensemble

### **4. Random Forest Regressor**

Model Random Forest Regressor adalah model ensemble yang terdiri dari multiple decision trees, sehingga dapat menangani kompleksitas data dengan lebih baik.

**Kelebihan Random Forest Regressor:**
* Performa yang lebih baik dibandingkan single decision tree
* Mengurangi risiko overfitting
* Dapat menangani fitur dengan berbagai skala
* Memberikan feature importance

**Kekurangan Random Forest Regressor:**
* Lebih kompleks dan membutuhkan lebih banyak resources
* Lebih sulit diinterpretasi dibandingkan single decision tree
* Training time yang lebih lama

### **5. Hyperparameter Tuning**

Untuk meningkatkan performa model Random Forest Regressor, dilakukan hyperparameter tuning menggunakan GridSearchCV dengan parameter grid berikut:

```
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

Hasil grid search menunjukkan parameter terbaik untuk model Random Forest adalah:
- n_estimators = 200
- max_depth = 20
- min_samples_split = 2
- min_samples_leaf = 1

### **6. Feature Importance**

Analisis feature importance dari model Random Forest menunjukkan kontribusi setiap fitur terhadap prediksi harga rumah. Fitur-fitur dengan importance tertinggi adalah:
1. sqft_living (luas bangunan)
2. grade (kualitas konstruksi)
3. lat (latitude/lokasi)
4. sqft_above (luas di atas tanah)
5. view (pemandangan)

Ini mengkonfirmasi bahwa luas rumah, kualitas konstruksi, dan lokasi adalah faktor-faktor utama yang mempengaruhi harga rumah.

## **Evaluation**

Untuk mengevaluasi performa model, beberapa metrik evaluasi diterapkan:

### **1. Mean Absolute Error (MAE)**

MAE mengukur rata-rata dari selisih absolut antara nilai aktual dan nilai prediksi.

![Image](https://github.com/user-attachments/assets/ce399d16-187f-4f6e-85cd-b49c84d746b2)

Dimana:
* n adalah jumlah sampel
* yi adalah nilai aktual
* ŷi adalah nilai prediksi

MAE memberikan informasi tentang seberapa besar kesalahan prediksi dalam unit yang sama dengan variabel target (dalam kasus ini, dollar).

### **2. Mean Squared Error (MSE)**

MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi.

![Image](https://github.com/user-attachments/assets/57dffca3-27a3-48d0-beb2-217010c9eb2a)

MSE memberikan bobot lebih pada error yang besar karena error dikuadratkan.

### **3. Root Mean Squared Error (RMSE)**

RMSE adalah akar kuadrat dari MSE. Ini memberikan estimasi standar deviasi dari error prediksi.

![Image](https://github.com/user-attachments/assets/1da0d0ba-c20d-47eb-8455-5a3f4ec3d375)

RMSE memiliki unit yang sama dengan variabel target dan lebih mudah diinterpretasi dibandingkan MSE.

### **4. R-squared (R²)**

R-squared mengukur proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh variabel independen dalam model.

![Image](https://github.com/user-attachments/assets/fc26c100-d2d2-488e-88e9-ea2ef14852a8)

Dimana:
* ȳ adalah mean dari nilai aktual
* R² bernilai antara 0 dan 1, dimana nilai yang lebih tinggi menunjukkan model yang lebih baik.

### **Hasil Evaluasi**

Berikut adalah hasil evaluasi dari tiga model yang diimplementasikan:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | $43,414.28 | $61,251.84 | 0.9099 |
| Decision Tree | $6,696.88 | $12,810.85 | 0.9961 |
| Random Forest (Tuned) | $2,658.26 | $7,385.04 | 0.9987 |

Model Random Forest dengan hyperparameter tuning memberikan performa terbaik dengan nilai RMSE terendah ($7,385.04) dan R² tertinggi (0.9987). Ini berarti model dapat menjelaskan sekitar 99.87% variasi dalam harga rumah, serta memiliki rata-rata error prediksi sebesar $7,385.04.

Mengingat rata-rata harga rumah dalam dataset adalah sekitar $540,088, maka RMSE dari model final hanya sekitar 1.37% dari rata-rata harga, yang menunjukkan bahwa model telah melampaui target awal untuk memiliki error kurang dari 20% dari rata-rata harga rumah.

### **Pengujian Model dengan Contoh Data**

Untuk menguji efektivitas model dalam kasus nyata, dilakukan prediksi harga pada sebuah sampel rumah dengan karakteristik berikut:

```
- Kamar tidur: 3
- Kamar mandi: 2.0
- Sqft living: 2000
- Sqft lot: 5000
- Lantai: 1.0
- Waterfront: 0
- View: 0
- Kondisi: 3
- Grade: 7
- Sqft above: 1500
- Sqft basement: 500
- Tahun dibangun: 1990
- Tahun renovasi: 0
- Zipcode: 98002
- Latitude: 47.5112
- Longitude: -122.257
- Sqft living15: 1800
- Sqft lot15: 4000
- Harga aktual: $695,000
```

Hasil prediksi menunjukkan:
- Harga prediksi: $679,769
- Selisih absolut: $15,231
- Persentase error: 2.19%

Persentase error yang sangat kecil (2.19%) menunjukkan bahwa model memiliki akurasi yang sangat baik dalam memprediksi harga rumah yang belum pernah dilihat sebelumnya.

## **Kesimpulan**

Pada proyek prediksi harga rumah ini, kami telah mengembangkan model machine learning untuk memperkirakan harga rumah berdasarkan berbagai fitur properti. Beberapa kesimpulan utama dari proyek ini:

1. **Performa Model**:
   - Model Random Forest dengan parameter teroptimasi memiliki performa terbaik dengan R² = 0.9987 dan RMSE = $7,385.04
   - Hal ini menunjukkan bahwa model mampu menjelaskan 99.87% variasi dalam harga rumah dengan error rata-rata sekitar $7,385
   - Uji prediksi pada sampel rumah menunjukkan error hanya 2.19%, membuktikan akurasi model yang sangat baik

2. **Fitur Penting**:
   - Luas rumah (sqft_living) adalah prediktor terkuat untuk harga rumah
   - Faktor lain yang signifikan meliputi grade (kualitas konstruksi), lokasi (lat, long), dan sqft_above (luas di atas tanah)
   - Faktor lokasi (zipcode) terbukti sangat berpengaruh terhadap harga properti

3. **Insight Bisnis**:
   - Luas dan kualitas konstruksi rumah merupakan faktor utama penentu harga
   - Lokasi properti tetap menjadi salah satu faktor terpenting dalam penilaian harga rumah
   - Renovasi memiliki dampak positif terhadap harga, meskipun tidak sebesar faktor luas dan lokasi

4. **Keterbatasan dan Pengembangan Masa Depan**:
   - Model saat ini belum mempertimbangkan faktor eksternal seperti kondisi ekonomi atau tren pasar properti
   - Penambahan data time-series untuk melacak perubahan harga properti dari waktu ke waktu dapat meningkatkan akurasi prediksi
   - Eksplorasi model deep learning dapat dipertimbangkan untuk meningkatkan kemampuan prediksi

Model prediksi harga rumah ini dapat digunakan oleh berbagai pemangku kepentingan di industri properti, termasuk agen real estate, pengembang properti, dan calon pembeli rumah. Keakuratan model memungkinkan estimasi harga yang lebih tepat, membantu pengambilan keputusan terkait investasi properti, dan memberikan pemahaman yang lebih baik tentang faktor-faktor yang mempengaruhi harga rumah di area yang diteliti.
