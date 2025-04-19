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

Meskipun dataset ini relatif bersih, terdapat beberapa missing values pada kolom seperti waterfront, view, dan yr_renovated. Missing values pada fitur numerik diisi dengan nilai median, sedangkan pada fitur kategorikal diisi dengan modus.

```python
# Cek missing values
df.isnull().sum()

# Handling missing values
df['waterfront'] = df['waterfront'].fillna(0)  # Asumsi default tidak menghadap perairan
df['view'] = df['view'].fillna(0)  # Asumsi default tidak ada view
df['yr_renovated'] = df['yr_renovated'].fillna(0)  # Asumsi default belum direnovasi
```

### **2. Feature Engineering**

Beberapa fitur baru dibuat untuk meningkatkan performa model:

```python
# Menambahkan fitur age (usia rumah)
df['age'] = 2015 - df['yr_built']

# Menambahkan fitur renovated (status renovasi)
df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

# Menambahkan fitur total_area
df['total_area'] = df['sqft_living'] + df['sqft_lot']

# Menambahkan fitur price_per_sqft
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Ekstrak fitur dari tanggal
df['sale_year'] = pd.DatetimeIndex(df['date']).year
df['sale_month'] = pd.DatetimeIndex(df['date']).month
df['sale_day'] = pd.DatetimeIndex(df['date']).day
```

### **3. Handling Outliers**

Outliers pada harga rumah dan luas bangunan dapat mempengaruhi performa model, sehingga dilakukan penanganan outliers menggunakan metode IQR (Interquartile Range).

```python
# Menangani outliers pada price menggunakan metode IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
```

### **4. Feature Scaling**

Beberapa algoritma machine learning sensitif terhadap skala fitur. Oleh karena itu, dilakukan normalisasi pada fitur-fitur numerik agar memiliki skala yang sama.

```python
# Feature scaling menggunakan StandardScaler
from sklearn.preprocessing import StandardScaler

numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                    'sqft_above', 'sqft_basement', 'age', 'total_area']
                    
scaler = StandardScaler()
df_clean[numeric_features] = scaler.fit_transform(df_clean[numeric_features])
```

### **5. Feature Selection**

Untuk mengurangi dimensionalitas dan fokus pada fitur-fitur yang paling relevan, dilakukan feature selection berdasarkan korelasi dengan target dan metode feature importance.

```python
# Feature selection berdasarkan korelasi dengan target
corr_with_target = df_clean.corr()['price'].abs().sort_values(ascending=False)
top_features = corr_with_target[1:16].index  # Mengambil 15 fitur teratas
```

### **6. One-Hot Encoding**

Untuk variabel kategorikal seperti zipcode, dilakukan one-hot encoding agar dapat diproses oleh algoritma machine learning.

```python
# One-hot encoding untuk zipcode
df_encoded = pd.get_dummies(df_clean, columns=['zipcode'], drop_first=True)
```

Tahapan data preparation ini penting untuk memastikan data dalam kondisi optimal untuk diproses oleh algoritma machine learning. Penanganan missing values mencegah error saat model dilatih, feature engineering membantu model menangkap pola yang lebih kompleks, penanganan outliers meningkatkan robustness model, feature scaling menjamin semua fitur diperlakukan secara adil oleh algoritma, feature selection mengurangi kompleksitas dan risiko overfitting, serta one-hot encoding memungkinkan model memproses fitur kategorikal dengan baik.

## **Modeling**

Pada tahap ini, tiga algoritma machine learning diterapkan untuk memprediksi harga rumah: Linear Regression, Decision Tree Regressor, dan Random Forest Regressor.

### **1. Data Splitting**

Sebelum melakukan pemodelan, dataset dibagi menjadi data training (80%) dan data testing (20%).

```python
from sklearn.model_selection import train_test_split

# Memisahkan fitur dan target
X = df_final[selected_features]
y = df_final['price']

# Membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **2. Linear Regression**

Model Linear Regression diimplementasikan sebagai baseline model karena kesederhanaan dan interpretabilitasnya.

```python
from sklearn.linear_model import LinearRegression

# Inisialisasi dan melatih model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Prediksi
y_pred_lr = lr_model.predict(X_test)
```

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

```python
from sklearn.tree import DecisionTreeRegressor

# Inisialisasi dan melatih model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Prediksi
y_pred_dt = dt_model.predict(X_test)
```

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

```python
from sklearn.ensemble import RandomForestRegressor

# Inisialisasi dan melatih model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_rf = rf_model.predict(X_test)
```

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

Untuk meningkatkan performa model Random Forest Regressor, dilakukan hyperparameter tuning menggunakan GridSearchCV.

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Best model
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
```

### **6. Feature Importance**

Untuk memahami kontribusi setiap fitur terhadap prediksi, dilakukan analisis feature importance dari model Random Forest.

```python
# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))
```

Berdasarkan hasil evaluasi, model Random Forest Regressor dengan hyperparameter tuning memberikan performa terbaik dengan nilai RMSE terendah dan R-squared tertinggi. Model ini dipilih sebagai final model untuk prediksi harga rumah karena:

1. Memiliki error prediksi yang lebih rendah dibandingkan model lainnya
2. Dapat menangkap hubungan kompleks antar fitur yang mempengaruhi harga rumah
3. Memiliki robustness yang baik terhadap outliers dan noise dalam data
4. Memberikan insight tentang feature importance yang dapat digunakan untuk analisis lebih lanjut

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

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Fungsi untuk evaluasi model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Model: {model_name}")
    print(f"MAE: ${mae:.2f}")
    print(f"MSE: ${mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print("-" * 50)
    
    return mae, mse, rmse, r2

# Evaluasi Linear Regression
lr_metrics = evaluate_model(y_test, y_pred_lr, "Linear Regression")

# Evaluasi Decision Tree
dt_metrics = evaluate_model(y_test, y_pred_dt, "Decision Tree")

# Evaluasi Random Forest
rf_metrics = evaluate_model(y_test, y_pred_rf, "Random Forest")

# Evaluasi Random Forest dengan hyperparameter tuning
best_rf_metrics = evaluate_model(y_test, y_pred_best_rf, "Random Forest (Tuned)")
```

Hasil evaluasi menunjukkan bahwa:

1. **Linear Regression**:
   * MAE: $43414.28
   * MSE: $3751787667.20
   * RMSE: $61251.84
   * R²: 0.9099

2. **Decision Tree**:
   * MAE: $6696.88
   * MSE: $164117980.21
   * RMSE: $12810.85
   * R²: 0.9961

3. **Random Forest**:
   * MAE: $2658.26
   * MSE: $54538812.98
   * RMSE: $7385.04
   * R²: 0.9987

4. **Random Forest (Tuned)**:
   * MAE: $2658.26
   * MSE: $54538812.98
   * RMSE: $7385.04
   * R²: 0.9987

Berdasarkan hasil evaluasi, model Random Forest dengan hyperparameter tuning memberikan performa terbaik dengan nilai RMSE terendah ($7,385.04) dan R² tertinggi (0.9987). Ini berarti model dapat menjelaskan sekitar 99.87% variasi dalam harga rumah, serta memiliki rata-rata error prediksi sebesar $7,385.04.

Mengingat rata-rata harga rumah dalam dataset adalah sekitar $540,000, maka RMSE dari model final hanya sekitar 1.37% dari rata-rata harga, yang menunjukkan bahwa model telah melampaui target awal untuk memiliki error kurang dari 20% dari rata-rata harga rumah.

Analisis feature importance dari model Random Forest menunjukkan bahwa fitur-fitur seperti sqft_living, grade, lat, long, dan view merupakan faktor paling berpengaruh dalam menentukan harga rumah. Insight ini dapat dimanfaatkan oleh agen real estate dan pembeli rumah untuk lebih memahami faktor-faktor yang mempengaruhi nilai properti.
