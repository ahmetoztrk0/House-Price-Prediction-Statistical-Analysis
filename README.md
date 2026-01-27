# House Price Prediction: İstatistiksel Modelleme ve Regresyon Analizi

Bu proje, Kaggle'ın "House Prices - Advanced Regression Techniques" veri seti üzerinde gerçekleştirilmiş; veri temizleme, keşifçi veri analizi (EDA) ve **Çoklu Doğrusal Regresyon** aşamalarını içeren kapsamlı bir istatistiksel çalışmadır.

## Projenin Amacı
Konut fiyatlarını etkileyen temel faktörleri belirlemek ve istatistiksel varsayımları (normallik, homoscedasticity vb.) koruyarak yüksek açıklayıcılık gücüne sahip bir tahmin modeli geliştirmektir.

---

## Teknik Metodoloji

### 1. Veri Ön İşleme ve İstatistiksel İmputasyon
* **Eksik Değer Yönetimi:** Bazı değişkenlerdeki "NaN" değerlerinin "yokluk" (Örn: Havuz yok, garaj yok) anlamına geldiği saptanarak uygun etiketleme yapılmıştır.
* **Grup Bazlı İmputasyon:** Arsa cephesi (`LotFrontage`) değişkeni, verinin dağılımını korumak adına mahalle bazlı medyan değerleri ile doldurulmuştur.

### 2. Varsayım Kontrolleri ve Veri Mühendisliği
* **Normalleştirme:** Bağımlı değişkenin (`SalePrice`) sağa çarpık dağılım sergilediği görülmüş ve modelin normallik varsayımını sağlamak amacıyla **Logaritmik Dönüşüm** (`log1p`) uygulanmıştır.
* **Aykırı Değer (Outlier) Analizi:** Serpilme diyagramları üzerinden yapılan analizle, modelin genel eğilimini bozan istatistiksel uç değerler veri setinden arındırılmıştır.



### 3. Modelleme ve Yorumlama (OLS Regression)
* **Değişken Seçimi:** Isı haritası (Heatmap) ve p-değerleri ($p < 0.05$) baz alınarak istatistiksel olarak anlamlı olan bağımsız değişkenler seçilmiştir.
* **İstatistiksel Analiz:** Model başarısı, **Adjusted R-squared** ve **Prob (F-statistic)** değerleri üzerinden test edilmiştir.


---

## 💻 Kullanılan Teknolojiler
* **Dil:** Python
* **Kütüphaneler:** Pandas, Numpy, Seaborn, Matplotlib, Statsmodels, Scikit-learn, Scipy

---

**Hazırlayan:** Ahmet Yaşar Öztürk - YTÜ İstatistik Bölümü Lisans Öğrencisi
